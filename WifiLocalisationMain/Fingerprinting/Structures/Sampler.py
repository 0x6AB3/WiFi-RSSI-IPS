import serial
import threading

# Starts a thread for recording RSSI values from a single IPS ESP32 running the 'Client code RSSI.ino' Arduino code
# Only takes samples to a certain limit (sample_count) before stopping the sampling thread
class Sampler:
    def __init__(self, port, sample_count=200):
        self.port = port # The COM_PORT value
        self.sample_count = sample_count
        self.rssi = []
        self.prepared = False # Buffer requires flushing due to boot messages and old values
        self.serial_reader = serial.Serial(f'COM{self.port}', 115200, timeout=1)
        print(f"COM_PORT{self.port}: Connected")

        self.sampling_thread = threading.Thread(target=self.thread_loop)

    def __len__(self):
        return len(self.rssi)

    def start_sampling_thread(self):
        if not self.prepared:
            print(f"COM_PORT{self.port} cannot take samples because it has not been prepared")
            return

        self.sampling_thread.start()
        print(f"COM_PORT{self.port}: Sampling started")

    # Reads a single RSSI value from the IPS node
    # If the data is not a valid RSSI value
    def get_single(self):
        if self.serial_reader.in_waiting > 0:
            try:
                data = self.serial_reader.readline().decode('utf-8').strip()
                data = int(data)
                return data
            except:
                pass
        return None

    # The ESP32 writes certain boot information which will cause errors if read as an RSSI integer
    # This method waits for this information to clear before reading RSSI
    # Errors may still occur if an IPS node is restarted due to the boot info being written to serial
    def prepare_for_sampling(self):
        if not self.prepared:
            self.rssi = []
            self.prepared = False
            self.sampling_thread = threading.Thread(target=self.thread_loop)
            self.serial_reader.close()
            self.serial_reader.open()

        print(f"COM_PORT{self.port}: Awaiting RSSI values")
        while not self.prepared:
            rssi = self.get_single()
            if rssi is not None and rssi <= 0:  # Begin reading when a negative integer is read
                self.prepared = True
                print(f"COM_PORT{self.port}: Ready for sampling RSSI")
            #print(f"Invalid value in thread {self.port - 4} COM_PORT{self.port}: {data}")

    def thread_loop(self):
        while len(self) < self.sample_count:
            rssi = self.get_single()
            if rssi is not None and rssi <= 0:
                self.rssi.append(rssi)
        self.prepared = False