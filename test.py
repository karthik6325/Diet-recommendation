import gzip
import csv

# Specify the paths for the compressed and uncompressed files
compressed_file_path = './dataset.csv'
uncompressed_file_path = './newdataset.csv'

# Open the compressed file and read its contents
with gzip.open(compressed_file_path, 'rb') as compressed_file:
    # Read the contents of the compressed file
    compressed_content = compressed_file.read()

# Write the binary contents to a new uncompressed CSV file
with open(uncompressed_file_path, 'wb') as uncompressed_file:
    # Write the contents to the new uncompressed file
    uncompressed_file.write(compressed_content)

print("Uncompressed CSV file created successfully.")