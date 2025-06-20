#include <WiFi.h>
#include <ESPping.h>

const char* ssid = "ESP32-Reference-Node";
const char* password = "precision";
IPAddress APIP;

void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  APIP = WiFi.gatewayIP();

  Serial.print("\nConnected to Reference Node (");
  Serial.print(APIP);
  Serial.println(")");
}

void loop() {
  int rssi = WiFi.RSSI();
  
  Serial.print("RSSI: ");
  Serial.print(rssi);
  Serial.println(" dBm");

  if (Ping.ping(APIP) > 0){
    Serial.printf(" response time : %d/%.2f/%d ms\n", Ping.minTime(), Ping.averageTime(), Ping.maxTime());
  } 
}
