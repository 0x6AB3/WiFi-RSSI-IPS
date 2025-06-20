#include <WiFi.h>

const char ssid[] = "ESP32-Reference-Node";
const char password[] = "precision";
WiFiServer server(80);

void setup() {
  Serial.begin(115200);

  WiFi.mode(WIFI_MODE_AP);
  WiFi.softAP(ssid, password);
  Serial.println("Reference node soft-AP initiated");
  
  server.begin();
  Serial.println("Reference node WiFi server initiated");

  Serial.print("AP IP: ");
  Serial.println(WiFi.softAPIP());
}

void loop() {
  WiFiClient client = server.available();
}