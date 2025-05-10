# Given a shapefile, split into a mangrove shapefile and a nonmangrove shapefile

import fiona
import os
import argparse

def split_vector(filepath):

	dir_path = os.path.dirname(filepath)

    # Creating necessary directories
	m_dir = os.path.join(dir_path, "m")
	nm_dir = os.path.join(dir_path, "nm")
	if os.path.exists(m_dir):
		os.system("rm -r " + m_dir)
	if os.path.exists(nm_dir):
		os.system("rm -r " + nm_dir)
	os.mkdir(m_dir)
	os.mkdir(nm_dir)

	with fiona.open(filepath) as source:
		m_file = os.path.join(m_dir, "m.shp")
		# Joining mangrove polygons
		with fiona.open(m_file, "w", crs=source.crs, driver=source.driver, schema=source.schema) as m_shp:
			print("Creating mangrove files in " + m_dir)
			for record in source:
				# Checking the there is in fact a class label
				try:
					if (record['properties']['class'] == "mangrove"):
						m_shp.write(record)
				except KeyError:
					try:
						if (record['properties']['Class'] == "mangrove"):
							m_shp.write(record)
					except KeyError:
						print("No class label... Exiting.")
						exit()

			print("Joined {} mangrove polygons.".format(len(m_shp)))

		nm_file = os.path.join(nm_dir, "nm.shp")
		# Joining nonmangrove polygons
		with fiona.open(nm_file, "w", crs=source.crs, driver=source.driver, schema=source.schema) as nm_shp:
			print("Creating nonmangrove files in " + nm_dir)
			for record in source:
				# Checking the there is in fact a class label
				try:
					if (record['properties']['class'] == "nonmangrove"):
						nm_shp.write(record)
				except KeyError:
					try:
						if (record['properties']['Class'] == "nonmangrove"):
							nm_shp.write(record)
					except KeyError:
						print("No class label... Exiting.")
						exit()
			print("Joined {} nonmangrove polygons.".format(len(nm_shp)))

if __name__ == "__main__":
	# Parser to specify which .shp file to split
	parser = argparse.ArgumentParser(description='Specify .shp file to be split')
	parser.add_argument('--filepath')

	args = parser.parse_args()
	filepath = args.file_name

	split_vector(filepath)