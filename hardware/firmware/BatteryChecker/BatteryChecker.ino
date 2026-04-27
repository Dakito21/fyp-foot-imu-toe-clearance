#include <Arduino.h>
#include <Adafruit_TinyUSB.h>
#include <bluefruit.h>

BLEUart bleuart;

static const uint8_t PIN_ADC_BAT_CTRL = 14;
static const uint8_t PIN_BAT_ADC      = 32;
static const uint8_t PIN_CHARGE_STAT  = 23;
static const uint8_t PIN_CHG_CURRENT  = 22;

static const bool USE_50MA_CHARGE = false;

static const float ADC_REF_V = 3.6f;
static const float ADC_MAX = 4095.0f;
static const float BAT_DIVIDER_FACTOR = 2.961f;

int trackedPct = -1;
bool lastCharging = false;
bool firstBatteryPrint = true;

void printBoth(const String& s) {
  Serial.println(s);
  if (Bluefruit.connected()) {
    bleuart.println(s);
  }
}

void startAdv() {
  Bluefruit.Advertising.stop();
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(bleuart);
  Bluefruit.ScanResponse.addName();
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(32, 244);
  Bluefruit.Advertising.setFastTimeout(30);
  Bluefruit.Advertising.start(0);
}

void setChargeCurrent() {
  if (USE_50MA_CHARGE) {
    pinMode(PIN_CHG_CURRENT, INPUT);
  } else {
    pinMode(PIN_CHG_CURRENT, OUTPUT);
    digitalWrite(PIN_CHG_CURRENT, LOW);
  }
}

bool isCharging() {
  pinMode(PIN_CHARGE_STAT, INPUT);
  return digitalRead(PIN_CHARGE_STAT) == LOW;
}

float readBatteryVoltage() {
  pinMode(PIN_ADC_BAT_CTRL, OUTPUT);
  digitalWrite(PIN_ADC_BAT_CTRL, LOW);
  delay(5);

  analogRead(PIN_BAT_ADC);
  delay(2);

  const int N = 8;
  uint32_t sum = 0;

  for (int i = 0; i < N; i++) {
    sum += analogRead(PIN_BAT_ADC);
    delay(2);
  }

  float raw = sum / (float)N;
  return BAT_DIVIDER_FACTOR * ADC_REF_V * raw / ADC_MAX;
}

int batteryPercent(float vbat) {
  if (vbat >= 4.20f) return 100;
  if (vbat <= 3.20f) return 0;

  int pct;

  if (vbat >= 4.00f) {
    pct = 80 + (int)((vbat - 4.00f) / 0.20f * 20.0f);
  } else if (vbat >= 3.85f) {
    pct = 55 + (int)((vbat - 3.85f) / 0.15f * 25.0f);
  } else if (vbat >= 3.70f) {
    pct = 25 + (int)((vbat - 3.70f) / 0.15f * 30.0f);
  } else if (vbat >= 3.50f) {
    pct = 8 + (int)((vbat - 3.50f) / 0.20f * 17.0f);
  } else {
    pct = (int)((vbat - 3.20f) / 0.30f * 8.0f);
  }

  return constrain(pct, 0, 100);
}

String batteryStatusText(bool charging, float vbat) {
  if (charging) return "C";
  if (vbat >= 4.20f) return "NC / Full";
  return "NC";
}

void setup() {
  Serial.begin(115200);
  delay(1200);

  analogReference(AR_DEFAULT);
  analogReadResolution(12);

  setChargeCurrent();

  Bluefruit.begin();
  Bluefruit.setTxPower(4);
  Bluefruit.setName("XIAO_BATTERY");

  bleuart.begin();
  startAdv();

  printBoth("XIAO Battery Monitor Ready");
}

void loop() {
  bool charging = isCharging();
  float vbat = readBatteryVoltage();
  int pct = batteryPercent(vbat);

  bool chargingChanged = !firstBatteryPrint && (charging != lastCharging);
  bool shouldPrint = false;

  if (firstBatteryPrint || chargingChanged) {
    trackedPct = pct;
    shouldPrint = true;
  } 
  else if (charging && pct > trackedPct) {
    trackedPct = pct;
    shouldPrint = true;
  } 
  else if (!charging && pct < trackedPct) {
    trackedPct = pct;
    shouldPrint = true;
  }

  if (shouldPrint) {
    printBoth(
      "S: " + batteryStatusText(charging, vbat) +
      " | B: " + String(vbat, 2) + "V, " +
      String(trackedPct) + "%"
    );
  }

  lastCharging = charging;
  firstBatteryPrint = false;

  while (bleuart.available()) {
    char c = bleuart.read();

    if (c == 'b' || c == 'B') {
      printBoth(
        "S: " + batteryStatusText(charging, vbat) +
        " | B: " + String(vbat, 2) + "V, " +
        String(pct) + "%"
      );
    }
  }

  delay(2000);
}