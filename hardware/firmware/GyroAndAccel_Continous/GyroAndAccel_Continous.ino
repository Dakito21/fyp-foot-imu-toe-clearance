#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_TinyUSB.h>
#include <bluefruit.h>

BLEUart bleuart;

const uint8_t BMI160_ADDR = 0x69;

// Registers
const uint8_t REG_CHIP_ID    = 0x00;
const uint8_t REG_PMU_STATUS = 0x03;
const uint8_t REG_GYR_DATA   = 0x0C;
const uint8_t REG_ACC_DATA   = 0x12;
const uint8_t REG_ACC_CONF   = 0x40;
const uint8_t REG_ACC_RANGE  = 0x41;
const uint8_t REG_GYR_CONF   = 0x42;
const uint8_t REG_GYR_RANGE  = 0x43;
const uint8_t REG_CMD        = 0x7E;

// Commands
const uint8_t CMD_SOFTRESET  = 0xB6;
const uint8_t CMD_ACC_NORMAL = 0x11;
const uint8_t CMD_GYR_NORMAL = 0x15;

// Settings
const uint8_t ACC_CONF_100HZ   = 0x28; // 100 Hz
const uint8_t ACC_RANGE_2G     = 0x03; // +/-2g
const uint8_t GYR_CONF_100HZ   = 0x28; // 100 Hz
const uint8_t GYR_RANGE_250DPS = 0x03; // +/-250 dps

static inline int16_t s16(uint8_t lo, uint8_t hi) {
  return (int16_t)((hi << 8) | lo);
}

void printSerialAndBLE(const String& s) {
  Serial.println(s);

  if (Bluefruit.connected()) {
    bleuart.println(s);
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
  Bluefruit.Advertising.setInterval(32, 244);
  Bluefruit.Advertising.setFastTimeout(30);
  Bluefruit.Advertising.start(0);
}

bool writeReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(BMI160_ADDR);
  Wire.write(reg);
  Wire.write(value);
  return (Wire.endTransmission() == 0);
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

bool initBMI160() {
  uint8_t chip = readReg(REG_CHIP_ID);

  String chipMsg = "CHIP_ID=0x";
  if (chip < 16) chipMsg += "0";
  chipMsg += String(chip, HEX);
  chipMsg.toUpperCase();
  printSerialAndBLE(chipMsg);

  if (chip != 0xD1) {
    printSerialAndBLE("BMI160 not detected");
    return false;
  }

  if (!writeReg(REG_CMD, CMD_SOFTRESET)) {
    printSerialAndBLE("Soft reset write failed");
    return false;
  }
  delay(100);

  writeReg(REG_CMD, CMD_ACC_NORMAL);
  delay(50);

  writeReg(REG_CMD, CMD_GYR_NORMAL);
  delay(100);

  writeReg(REG_ACC_CONF, ACC_CONF_100HZ);
  writeReg(REG_ACC_RANGE, ACC_RANGE_2G);
  writeReg(REG_GYR_CONF, GYR_CONF_100HZ);
  writeReg(REG_GYR_RANGE, GYR_RANGE_250DPS);
  delay(50);

  uint8_t pmu = readReg(REG_PMU_STATUS);
  String pmuMsg = "PMU_STATUS=0x";
  if (pmu < 16) pmuMsg += "0";
  pmuMsg += String(pmu, HEX);
  pmuMsg.toUpperCase();
  printSerialAndBLE(pmuMsg);

  return true;
}

void setup() {
  Serial.begin(115200);
  delay(1200);

  Wire.begin();

  Bluefruit.begin();
  Bluefruit.setTxPower(4);
  Bluefruit.setName("XIAO_BATTERY");

  bleuart.begin();
  bleuart.bufferTXD(true);   // important for more reliable BLE UART sending
  startAdv();

  printSerialAndBLE("");
  printSerialAndBLE("BMI160 BLE stream starting...");
  printSerialAndBLE("");

  initBMI160();
}

void loop() {
  uint8_t ab[6];
  uint8_t gb[6];

  bool okA = readRegs(REG_ACC_DATA, ab, 6);
  bool okG = readRegs(REG_GYR_DATA, gb, 6);

  if (!okA || !okG) {
    printSerialAndBLE("Read failed");
    if (Bluefruit.connected()) {
      bleuart.flushTXD();
    }
    delay(300);
    return;
  }

  int16_t ax = s16(ab[0], ab[1]);
  int16_t ay = s16(ab[2], ab[3]);
  int16_t az = s16(ab[4], ab[5]);

  int16_t gx = s16(gb[0], gb[1]);
  int16_t gy = s16(gb[2], gb[3]);
  int16_t gz = s16(gb[4], gb[5]);

  // BMI160 conversion
  float ax_g = ax / 16384.0f;
  float ay_g = ay / 16384.0f;
  float az_g = az / 16384.0f;

  float gx_dps = gx / 131.2f;
  float gy_dps = gy / 131.2f;
  float gz_dps = gz / 131.2f;

  // Send shorter messages instead of one long message
  String accLine = "ACC[g]: ";
  accLine += String(ax_g, 3);
  accLine += ", ";
  accLine += String(ay_g, 3);
  accLine += ", ";
  accLine += String(az_g, 3);

  String gyrLine = "GYR[dps]: ";
  gyrLine += String(gx_dps, 3);
  gyrLine += ", ";
  gyrLine += String(gy_dps, 3);
  gyrLine += ", ";
  gyrLine += String(gz_dps, 3);

  printSerialAndBLE(accLine);
  printSerialAndBLE(gyrLine);

  if (Bluefruit.connected()) {
    bleuart.flushTXD();
  }

  while (bleuart.available()) {
    char c = (char)bleuart.read();
    Serial.print("RX: ");
    Serial.println(c);
  }

  delay(300);
}