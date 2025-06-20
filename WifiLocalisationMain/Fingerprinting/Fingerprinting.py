import sys
import time
import matplotlib.pyplot as plt
from Structures.Samplers import Samplers

# My configuration forms an equilateral triangle ~340cm per side, 1.5m above ground as follows:
    #   COM_PORT4 = Reference node (running 'Soft_AP.ino' Arduino code on an ESP32)
    #   COM_PORT5 = IPS Node 1 (running RSSI_Client.ino' Arduino code on an ESP32 located at the radiator wall)
    #   COM_PORT6 = IPS Node 2 (running 'RSSI_Client.ino' Arduino code on an ESP32 located at the angled wall)
    #   COM_PORT7 = IPS Node 3 (running 'RSSI_Client.ino' Arduino code on an ESP32 located at the bed wall)

# Sampling loop for 3 IPS nodes
sample_count = 200  # Quantity of RSSI readings to measure per ESP32 in IPS
ports = [9, 11, 12]
samplers = Samplers(0, 0) # Creating IPS network

try:
    samplers.create_samplers(ports, 200) # Filling IPS network
except:
    print(f"ESP32 not found on all COMPORTs {ports}")
    sys.exit()

filename = "../Datasets/Fingerprinting.csv"  # File to store RSSI readings
command = ""
locations_sampled = 0
while command != "exit":
    print(f"\nLocations sampled: {locations_sampled}") # Total positions at which RSSI was sampled during this runtime
    print(f"Sampling {sample_count} times per IPS node")
    print(f"Current location (x,y): ({samplers.x}cm,{samplers.y}cm)") # Distance from COM_PORT5 (radiator wall)
    command = input("Press [ENTER] to sample, type 'location' to change position, or 'exit' to quit: ")
    # Changing reference node position
    if command == 'location':
        new_location = input("Enter new position with following format 'x,y' (where x and y are integers representing centimeters): ")
        x, y = new_location.split(",")[0], new_location.split(",")[1]
        samplers.change_location(x, y)
        print(f"Location changed to ({samplers.x}cm,{samplers.y}cm)")
    # Recording samples
    elif command == "":
        print("Run!!!")
        time.sleep(3)
        print("Stand still")
        samplers.start_sampling()
        # Plotting present samples for manual outlier detection
        plt.clf()
        colors = ["red", "green", "blue"]  # Colours for plotting RSSI samples from 3 samplers
        for sampler in samplers:
            plt.plot(sampler.rssi, label=f"COM_PORT{sampler.port}", color=colors.pop(0))
        plt.title(f"{sample_count*len(samplers)} RSSI samples at ({samplers.x}cm, {samplers.y}cm)")
        plt.xlabel("Sample index")
        plt.ylabel("RSSI (dBm)")
        plt.legend()
        plt.show()
        # Appending results to fingerprinting csv
        write = input(f"Write to {filename}? (y/n, default=y): ")
        if write != "n":
            samplers.write_to_file(filename)
            locations_sampled += 1