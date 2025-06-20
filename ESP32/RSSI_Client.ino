#include <WiFi.h>

const char ssid[] = "ESP32-Reference-Node";
const char password[] = "precision";

void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }

  Serial.println("Connected to Reference Node");
}

void loop() {
  Serial.print(WiFi.RSSI());
}
