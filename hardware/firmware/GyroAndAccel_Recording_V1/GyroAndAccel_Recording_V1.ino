#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_TinyUSB.h>
#include <bluefruit.h>

/*
  XIAO nRF52840 + external BMI160 over I2C

  Reliable method:
    1) Record raw IMU samples into XIAO RAM at 100 Hz.
    2) Python sends stop.
    3) XIAO dumps CSV in chunks.
    4) Python verifies every seq in the chunk.
    5) Python sends ACK,<next_seq> if complete, or NACK,<chunk_start> if incomplete.
    6) XIAO only sends the next chunk after ACK.

  This solves the previous problem where Python saved fewer CSV rows than the XIAO recorded.

  Python commands over BLE UART:
    start           -> start RAM recording
    stop            -> stop RAM recording and begin reliable CSV dump
    ACK,<next_seq>  -> Python confirms chunk received completely
    NACK,<seq>      -> Python requests resend from seq/chunk start
    cancel          -> cancel recording/dump
    status          -> print current state
*/

BLEUart bleuart;

static const char* DEVICE_NAME = "XIAO_BATTERY";

// Your I2C scanner showed BMI160 at 0x69. If it changes, set this to 0x68.
static const uint8_t BMI160_ADDR = 0x69;

// BMI160 registers
static const uint8_t REG_CHIP_ID    = 0x00;
static const uint8_t REG_GYR_DATA   = 0x0C;  // burst from 0x0C gives gyro then accel
static const uint8_t REG_ACC_CONF   = 0x40;
static const uint8_t REG_ACC_RANGE  = 0x41;
static const uint8_t REG_GYR_CONF   = 0x42;
static const uint8_t REG_GYR_RANGE  = 0x43;
static const uint8_t REG_CMD        = 0x7E;

static const uint8_t CMD_SOFTRESET  = 0xB6;
static const uint8_t CMD_ACC_NORMAL = 0x11;
static const uint8_t CMD_GYR_NORMAL = 0x15;

// BMI160 config: 100 Hz, normal/OSR4 bandwidth, +/-2g, +/-250 dps
static const uint8_t ACC_CONF_100HZ   = 0x28;
static const uint8_t ACC_RANGE_2G     = 0x03;
static const uint8_t GYR_CONF_100HZ   = 0x28;
static const uint8_t GYR_RANGE_250DPS = 0x03;

static const uint16_t SAMPLE_HZ = 100;
static const uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_HZ;

// 6000 samples = 60 seconds at 100 Hz.
// sizeof(ImuSample) is normally 16 bytes, so this uses about 96 KB RAM.
static const uint16_t MAX_SAMPLES = 6000;

// Reliable CSV dump settings.
// 20 rows per chunk keeps the CSV readable and limits retransmission size.
static const uint16_t CHUNK_LINES = 20;
static const uint32_t ACK_TIMEOUT_MS = 3000;
static const uint8_t MAX_CHUNK_RETRIES = 20;

// Conservative line-burst throttle inside each chunk.
static const uint8_t FLUSH_EVERY_LINES = 2;
static const uint8_t FLUSH_DELAY_MS = 2;

// Set 1 only when debugging over USB Serial. Leaving it 0 makes dump faster.
static const uint8_t SERIAL_ECHO_CSV = 0;

struct ImuSample {
  uint32_t t_ms;
  int16_t ax;
  int16_t ay;
  int16_t az;
  int16_t gx;
  int16_t gy;
  int16_t gz;
};

ImuSample samples[MAX_SAMPLES];

bool bmi_ok = false;
bool recording = false;
bool dumping = false;
bool waiting_for_ack = false;

uint32_t record_start_ms = 0;
uint32_t next_sample_us = 0;
uint16_t sample_count = 0;
uint16_t dump_index = 0;
uint32_t missed_samples = 0;
uint32_t read_failures = 0;

uint16_t current_chunk_start = 0;
uint16_t current_chunk_end = 0;  // exclusive
uint8_t chunk_retry_count = 0;
uint32_t last_chunk_send_ms = 0;

String cmd_buf = "";

static inline int16_t s16(uint8_t lo, uint8_t hi) {
  return (int16_t)((uint16_t)hi << 8 | lo);
}

bool writeReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(BMI160_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool readRegs(uint8_t reg, uint8_t* buf, size_t len) {
  Wire.beginTransmission(BMI160_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;

  size_t n = Wire.requestFrom(BMI160_ADDR, (uint8_t)len);
  if (n != len) return false;

  for (size_t i = 0; i < len; i++) {
    buf[i] = Wire.read();
  }
  return true;
}

uint8_t readReg(uint8_t reg) {
  uint8_t v = 0xFF;
  readRegs(reg, &v, 1);
  return v;
}

void flushBle() {
  if (Bluefruit.connected()) {
    bleuart.flushTXD();
  }
}

void bleWriteRaw(const char* text) {
  if (Bluefruit.connected()) {
    bleuart.write((const uint8_t*)text, strlen(text));
  }
}

void sendLine(const String& line) {
  Serial.println(line);
  if (Bluefruit.connected()) {
    String out = line + "\n";
    bleuart.write((const uint8_t*)out.c_str(), out.length());
  }
}

void startAdv() {
  Bluefruit.Advertising.stop();
  Bluefruit.ScanResponse.clearData();
  Bluefruit.Advertising.clearData();

  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(bleuart);
  Bluefruit.ScanResponse.addName();

  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(32, 244);  // 20 ms fast, 152.5 ms slow
  Bluefruit.Advertising.setFastTimeout(30);
  Bluefruit.Advertising.start(0);
}

bool initBMI160() {
  uint8_t chip = readReg(REG_CHIP_ID);

  String chipMsg = "CHIP_ID=0x";
  if (chip < 16) chipMsg += "0";
  chipMsg += String(chip, HEX);
  chipMsg.toUpperCase();
  sendLine(chipMsg);

  if (chip != 0xD1) {
    sendLine("ERR,BMI160_NOT_DETECTED_CHECK_ADDR_WIRING_EXPECT_0xD1");
    return false;
  }

  writeReg(REG_CMD, CMD_SOFTRESET);
  delay(120);

  writeReg(REG_CMD, CMD_ACC_NORMAL);
  delay(60);
  writeReg(REG_CMD, CMD_GYR_NORMAL);
  delay(120);

  bool ok = true;
  ok &= writeReg(REG_ACC_CONF, ACC_CONF_100HZ);
  ok &= writeReg(REG_ACC_RANGE, ACC_RANGE_2G);
  ok &= writeReg(REG_GYR_CONF, GYR_CONF_100HZ);
  ok &= writeReg(REG_GYR_RANGE, GYR_RANGE_250DPS);
  delay(50);

  if (!ok) {
    sendLine("ERR,BMI160_CONFIG_WRITE_FAILED");
    return false;
  }

  sendLine("BMI160_READY,100Hz,I2C_400kHz,ACC_2G,GYR_250DPS");
  return true;
}

bool readRaw(int16_t& ax, int16_t& ay, int16_t& az,
             int16_t& gx, int16_t& gy, int16_t& gz) {
  uint8_t b[12];
  if (!readRegs(REG_GYR_DATA, b, 12)) return false;

  gx = s16(b[0], b[1]);
  gy = s16(b[2], b[3]);
  gz = s16(b[4], b[5]);
  ax = s16(b[6], b[7]);
  ay = s16(b[8], b[9]);
  az = s16(b[10], b[11]);
  return true;
}

void captureOneSample() {
  if (sample_count >= MAX_SAMPLES) {
    missed_samples++;
    return;
  }

  ImuSample& s = samples[sample_count];

  if (!readRaw(s.ax, s.ay, s.az, s.gx, s.gy, s.gz)) {
    read_failures++;
    missed_samples++;
    return;
  }

  s.t_ms = millis() - record_start_ms;
  sample_count++;
}

void startRecording() {
  if (!bmi_ok) {
    sendLine("ERR,BMI160_NOT_READY");
    flushBle();
    return;
  }
  if (recording) {
    sendLine("ERR,ALREADY_RECORDING");
    flushBle();
    return;
  }
  if (dumping) {
    sendLine("ERR,DUMP_IN_PROGRESS");
    flushBle();
    return;
  }

  sample_count = 0;
  dump_index = 0;
  missed_samples = 0;
  read_failures = 0;

  current_chunk_start = 0;
  current_chunk_end = 0;
  chunk_retry_count = 0;
  waiting_for_ack = false;

  record_start_ms = millis();
  next_sample_us = micros();
  recording = true;

  sendLine("STREAM_STARTED,100Hz,RAM_THEN_CHUNKED_CSV_ACK");
  flushBle();
}

void finishDump() {
  dumping = false;
  waiting_for_ack = false;

  String stopped = "STREAM_STOPPED," + String(sample_count) +
                   " samples,missed=" + String(missed_samples) +
                   ",read_failures=" + String(read_failures);
  sendLine(stopped);
  flushBle();
  delay(20);

  // Repeat CSV_END a few times because final metadata has no ACK.
  // Python finalizes on the first CSV_END and ignores duplicates.
  sendLine("CSV_END");
  flushBle();
  delay(20);
  sendLine("CSV_END");
  flushBle();
  delay(20);
  sendLine("CSV_END");
  flushBle();
}

void sendCurrentChunk() {
  if (!dumping) return;

  if (current_chunk_start >= sample_count) {
    finishDump();
    return;
  }

  current_chunk_end = current_chunk_start + CHUNK_LINES;
  if (current_chunk_end > sample_count) current_chunk_end = sample_count;

  sendLine("CHUNK_BEGIN," + String(current_chunk_start) + "," + String(current_chunk_end));

  char line[128];
  uint8_t burst_lines = 0;

  for (uint16_t seq = current_chunk_start; seq < current_chunk_end; seq++) {
    const ImuSample& s = samples[seq];

    snprintf(line, sizeof(line),
             "%u,%lu,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n",
             seq,
             (unsigned long)s.t_ms,
             s.ax / 16384.0f,
             s.ay / 16384.0f,
             s.az / 16384.0f,
             s.gx / 131.2f,
             s.gy / 131.2f,
             s.gz / 131.2f);

    if (SERIAL_ECHO_CSV) Serial.print(line);
    bleWriteRaw(line);

    burst_lines++;
    if (burst_lines >= FLUSH_EVERY_LINES) {
      flushBle();
      delay(FLUSH_DELAY_MS);
      burst_lines = 0;
    }
  }

  flushBle();
  sendLine("CHUNK_END," + String(current_chunk_start) + "," + String(current_chunk_end));
  flushBle();

  waiting_for_ack = true;
  last_chunk_send_ms = millis();
}

void beginDump() {
  dumping = true;
  waiting_for_ack = false;
  dump_index = 0;
  current_chunk_start = 0;
  current_chunk_end = 0;
  chunk_retry_count = 0;

  sendLine("CSV_BEGIN");
  sendLine("seq,t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps");
  sendLine("DUMP_INFO,total=" + String(sample_count) + ",chunk=" + String(CHUNK_LINES));
  flushBle();
}

void stopRecordingAndDump() {
  if (!recording) {
    if (dumping) sendLine("ERR,ALREADY_DUMPING");
    else sendLine("ERR,NOT_RECORDING");
    flushBle();
    return;
  }

  recording = false;
  beginDump();
}

void cancelRecordingOrDump() {
  recording = false;
  dumping = false;
  waiting_for_ack = false;
  dump_index = 0;
  sample_count = 0;
  sendLine("CANCELLED");
  flushBle();
}

void handleAck(uint16_t next_seq) {
  if (!dumping || !waiting_for_ack) {
    sendLine("ERR,ACK_UNEXPECTED");
    flushBle();
    return;
  }

  if (next_seq == current_chunk_end) {
    dump_index = next_seq;
    current_chunk_start = next_seq;
    waiting_for_ack = false;
    chunk_retry_count = 0;
    return;
  }

  sendLine("ERR,ACK_MISMATCH,got=" + String(next_seq) +
           ",expected=" + String(current_chunk_end));
  flushBle();
}

void handleNack(uint16_t resend_seq) {
  if (!dumping) {
    sendLine("ERR,NACK_UNEXPECTED");
    flushBle();
    return;
  }

  // Resend the current chunk. If Python sends the exact chunk start, use it.
  // Otherwise clamp to the current chunk start to avoid jumping backwards too far.
  if (resend_seq == current_chunk_start || resend_seq == dump_index) {
    waiting_for_ack = false;
    chunk_retry_count++;
    if (chunk_retry_count > MAX_CHUNK_RETRIES) {
      sendLine("ERR,MAX_RETRIES_ABORT,chunk=" + String(current_chunk_start));
      flushBle();
      dumping = false;
      waiting_for_ack = false;
      return;
    }
    sendCurrentChunk();
  } else {
    sendLine("ERR,NACK_MISMATCH,got=" + String(resend_seq) +
             ",chunk_start=" + String(current_chunk_start));
    flushBle();
  }
}

int valueAfterComma(const String& cmd) {
  int comma = cmd.indexOf(',');
  if (comma < 0) return -1;
  return cmd.substring(comma + 1).toInt();
}

void handleCommand(String cmd) {
  cmd.trim();
  cmd.toLowerCase();
  if (cmd.length() == 0) return;

  Serial.print("RX_CMD=[");
  Serial.print(cmd);
  Serial.println("]");

  // Do not echo RX_CMD into BLE. Python expects a clean parseable stream.

  if (cmd == "start" || cmd == "r") {
    startRecording();
  }
  else if (cmd == "stop") {
    stopRecordingAndDump();
  }
  else if (cmd == "cancel" || cmd == "c") {
    cancelRecordingOrDump();
  }
  else if (cmd.startsWith("ack,")) {
    int v = valueAfterComma(cmd);
    if (v >= 0) handleAck((uint16_t)v);
    else sendLine("ERR,BAD_ACK");
  }
  else if (cmd.startsWith("nack,")) {
    int v = valueAfterComma(cmd);
    if (v >= 0) handleNack((uint16_t)v);
    else sendLine("ERR,BAD_NACK");
  }
  else if (cmd == "status") {
    if (recording) {
      sendLine("STATUS,RECORDING," + String(sample_count) + "/" + String(MAX_SAMPLES));
    } else if (dumping) {
      sendLine("STATUS,DUMPING,chunk=" + String(current_chunk_start) + "-" +
               String(current_chunk_end) + ",dump_index=" + String(dump_index) +
               ",total=" + String(sample_count) +
               ",waiting_ack=" + String(waiting_for_ack ? 1 : 0));
    } else {
      sendLine("STATUS,IDLE,last_samples=" + String(sample_count));
    }
    flushBle();
  }
  else {
    sendLine("ERR,UNKNOWN_CMD");
    flushBle();
  }
}

void readBleCommands() {
  while (bleuart.available()) {
    char c = (char)bleuart.read();
    if (c == '\n' || c == '\r') {
      handleCommand(cmd_buf);
      cmd_buf = "";
    } else {
      cmd_buf += c;
      if (cmd_buf.length() > 60) {
        cmd_buf = "";
        sendLine("ERR,CMD_TOO_LONG");
        flushBle();
      }
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1200);

  Wire.begin();
  Wire.setClock(400000);

  // Must be before Bluefruit.begin(). Increases BLE UART bandwidth resources.
  Bluefruit.configPrphBandwidth(BANDWIDTH_MAX);
  Bluefruit.begin();
  Bluefruit.setTxPower(4);
  Bluefruit.setName(DEVICE_NAME);

  // Request short connection interval: min 7.5 ms, max 15 ms.
  // The central device can still reject/adjust it.
  Bluefruit.Periph.setConnInterval(6, 12);

  bleuart.begin();
  bleuart.bufferTXD(true);  // Required for efficient small BLE UART writes.

  startAdv();

  sendLine("XIAO_RAM_100HZ_CHUNKED_CSV_ACK_READY");
  sendLine("Commands: start, stop, ACK,<next_seq>, NACK,<seq>, cancel, status");

  bmi_ok = initBMI160();
  flushBle();
}

void loop() {
  readBleCommands();

  if (recording) {
    uint32_t now_us = micros();

    while ((int32_t)(now_us - next_sample_us) >= 0) {
      next_sample_us += SAMPLE_PERIOD_US;
      captureOneSample();

      // Avoid a runaway catch-up loop if I2C/interrupts cause a large delay.
      if ((int32_t)(micros() - next_sample_us) > (int32_t)(5 * SAMPLE_PERIOD_US)) {
        missed_samples++;
        next_sample_us = micros() + SAMPLE_PERIOD_US;
        break;
      }
    }
  }

  if (dumping) {
    if (!waiting_for_ack) {
      sendCurrentChunk();
    } else if (millis() - last_chunk_send_ms > ACK_TIMEOUT_MS) {
      chunk_retry_count++;
      if (chunk_retry_count > MAX_CHUNK_RETRIES) {
        sendLine("ERR,ACK_TIMEOUT_ABORT,chunk=" + String(current_chunk_start));
        flushBle();
        dumping = false;
        waiting_for_ack = false;
      } else {
        sendLine("RETRY,chunk=" + String(current_chunk_start) + ",attempt=" + String(chunk_retry_count));
        flushBle();
        waiting_for_ack = false;
        sendCurrentChunk();
      }
    }
  }
}
