import os
from .Sampler import Sampler

# Container for many sampler objects representing an Indoor Positioning System (IPS)
# Provides methods for starting all samplers and appending their RSSI readings to a csv file for ML development
# Expects multiple ESP32 microcontrollers to each be running the 'RSSI_Client.ino' Arduino code
# Each ESP32 outputs the current RSSI of the COM_PORT4 reference node AP that runs the 'Soft_AP.ino' Arduino code
class Samplers:
    def __init__(self, x=0, y=0):
        self.samplers = []

        # The reference node location relative to IPS Node 1 on COM_PORT 5 (Radiator wall)
        # x = horizontal distance (along the line of sight from Node 1 --> Node 2)
        # y = vertical distance (distance from center of line [Node 1 --> Node 2] to Node 3)
        self.x = x
        self.y = y

    # Retrieve a single sampler
    def __getitem__(self, index):
        return self.samplers[index]

    def __len__(self):
        return len(self.samplers)

    # For taking RSSI readings at a new location without restarting/reinstantiation
    def change_location(self, x, y):
        self.x = x
        self.y = y

    # Add sampler manually
    def append(self, sampler):
        self.samplers.append(sampler)

    # Creating and containing a sequence of samplers for several ports
    def create_samplers(self, ports, sample_count):
        self.samplers = []
        for port in ports:
            sampler = Sampler(port, sample_count)
            self.append(sampler)

    def get_singles(self):
        return [sampler.get_single() for sampler in self.samplers]

    # Fill the RSSI list of each sampler with realtime values
    def start_sampling(self):

        # Exit if no samplers have been added/created
        if len(self.samplers) == 0:
            print("Error: no samplers found")
            return

        for sampler in self.samplers:
            sampler.prepare_for_sampling()

        # Starting samplers almost simultaneously to ensure a level of synchronisation
        for sampler in self.samplers:
            sampler.start_sampling_thread()
        print("All samplers started")

        # Waiting for the RSSI lists to fill up
        for sampler in self.samplers:
            sampler.sampling_thread.join()
            print(f"COM_PORT{sampler.port}: Sampling complete with RSSI sample count: {len(sampler.rssi)}")
        print("All samplers finished")

    # Saving recorded RSSI values for my specific configuration
    # NOTE: This code requires changes if using more/less than 3 IPS nodes/samplers
    def write_to_file(self, filename):

        # If RSSI sample counts are not equal, ML models may be affected due to introduction of bias towards nodes with more data
        equal_sample_count = len(self.samplers[0]) == len(self.samplers[1]) == len(self.samplers[2])
        if not equal_sample_count:
            print("Warning: uneven distribution of RSSI values; not all samplers have an equal sample count")

        # Precomputing lines to be written to the csv for reduction of file access time and system calls
        lines_to_write = []
        for sample_index in range(len(self.samplers[0])):
            lines_to_write.append(f"{self.x},{self.y},{self.samplers[0].rssi[sample_index]},{self.samplers[1].rssi[sample_index]},{self.samplers[2].rssi[sample_index]}\n")

        # File will require a new header if not yet created
        exists = os.path.isfile(filename)

        # Samples are appended so that fingerprinting may be carried out over several sessions due to large total sample count
        with open(filename, 'a') as file:
            if not exists:
                file.write(f"x,y,rssi_for_port{self.samplers[0].port},rssi_for_port{self.samplers[1].port},rssi_for_port{self.samplers[2].port}\n")
            file.writelines(lines_to_write)
            file.close()
