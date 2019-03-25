from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import datetime
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import ipywidgets
from IPython.display import display, clear_output, Javascript, Markdown

def subdirectories_of(path):
	return [item.resolve() for item in path.iterdir() if item.is_dir()]

def load_data():
	current_path = Path().resolve()
	git_path = current_path

	# While we are not at the root of the git directory:
	while not '.git' in map(lambda p: p.name, subdirectories_of(git_path)):
		# Move one directory up, and check again
		git_path = git_path.parent.resolve()
		if len(git_path.parts) <= 1:
			raise Warning("This script is not running in the git repository. Configure data path manually.")


	data_path = git_path / "data" / "origineel"

	if not Path(data_path).is_dir():
		raise Warning("Data path does not exist")
	data_file_paths = list(data_path.glob("*.csv"))

	if len(data_file_paths) == 0:
		raise Warning("No .csv files found in " + str(data_path) + "; Note that data files are not stored in this repository, and must be downloaded manually.")


	# Geef het circuitnummer (4 cijfers) dat in de naam van een Path staat
	circuitnum_from_file_path = lambda file_path: int(file_path.name.split('-')[0])

	# Drie dictionaries, met als keys de circuitnummers (int), als value de Pandas DataFrame van de .csv.
	cable_config       = {circuitnum_from_file_path(fp): pd.read_csv(fp, sep=";") for fp in data_file_paths if 'cableconfig' in fp.name}
	partial_discharges = {circuitnum_from_file_path(fp): pd.read_csv(fp, sep=";") for fp in data_file_paths if 'pd' in fp.name}
	warning            = {circuitnum_from_file_path(fp): pd.read_csv(fp, sep=";") for fp in data_file_paths if 'warning' in fp.name}

	# Sla de keys op van alle kabels waarvoor PD data bestaat.
	circuits = list(partial_discharges)

	return circuits, cable_config, partial_discharges, warning

def identify_suspicious_areas(pd, cable_length, bin_size=4.0, count_charges=False, lower_limit_nominal=80.0/100.0, lower_limit_fault=95.0/100.0):
	# Sla de drie kolomnamen van PD voor het gemak op
	datetimeC, locC, chargeC = pd.columns
	
	# Lijst met bools die aangeeft of er op dat tijdstip een PD was
	pd_occured = ~np.isnan(pd[locC])
	
	# De drie kolommen; alleen op tijdstippen met PD
	locations = pd[locC][pd_occured]
	charges = pd[chargeC][pd_occured]
	times = pd[datetimeC][pd_occured].apply(lambda date_string: datetime.datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S"))

	bins = np.arange(start=0., stop=cable_length, step=bin_size)
	# Tel inhoud van elk bakje:
	binned_pds, _ = np.histogram(locations, bins=bins, weights=charges if count_charges else None)
	
	# Reken het nominale level uit:
	nominal_pd_level = np.sort(binned_pds)[int(lower_limit_nominal * len(binned_pds))]
	
	# Bereken de parameter van de Poisson verdeling:
	square = lambda x: x*x

	phieta = scipy.stats.norm.ppf(q=lower_limit_nominal)
	rate = .25*square(-phieta + np.sqrt(square(phieta) + 4*nominal_pd_level))

	if rate == 0.0:
		# rate was zero: fault_pd_level should be zero, not nan
		fault_pd_level = 0.0
	else:
		# Als ik de ppf van de scipy.stats.poisson gebruik krijg ik alleen maar resultaten heel dichtbij M_{\eta}
		# dat hoort niet (toch?)
		fault_pd_level = scipy.stats.norm.ppf(q=lower_limit_fault, loc=rate, scale=rate)
	
	# Identificeer de probleemgebieden:
	problem_areas = binned_pds > fault_pd_level
	return problem_areas, locations, charges, times, bins

def cluster_boolean_series(series, max_consecutive_false=2, min_length=0, min_count=3):
	"""
	Imperative algorithm to identify sequences of mostly True values, under the conditions imposed by the parameters:
    
	Parameters
	----------
    
	series : array_like
		Sequence of Booleans to cluster
    
	max_consecutive_false : int
		Maximum number of consecutive Falsey values to accept inside a cluster
    
	min_length : int
		Minimum cluster length (right bound - left bound)
    
	min_count : int
		Minimum cluster size (number of Truthy values inside cluster bounds)
	"""
    
	clusters = set()
    
    
	cluster_start = 0   # Beginindex van het huidige cluster
	gap_size = 0        # Lengte van de rij nee'tjes die nu wordt belopen
	true_count = 0      # Aantal ja'tjes dat is gevonden in dit cluster
    
	for i, x in enumerate(series):
		if x:  # We doorlopen ja'tjes
			true_count += 1
			if gap_size > max_consecutive_false:   # Einde cluster
				cluster_end = i - gap_size
				if cluster_end - cluster_start >= min_length and true_count >= min_count:
					# Cluster was lang genoeg en heeft genoeg ja'tjes:
					clusters.add((cluster_start, cluster_end))
                
				# Begin een nieuw cluster
				cluster_start = i
				true_count = 0
            
			gap_size = 0  # We doorlopen geen nee'tjes (meer)
        
		if not x:  # We doorlopen nee'tjes
            		gap_size += 1
    
	cluster_end = len(series) - gap_size
	if cluster_end - cluster_start >= min_length and true_count >= min_count:
		# Cluster was lang genoeg en heeft genoeg ja'tjes:
		clusters.add((cluster_start, cluster_end))
	return clusters

def draw_shaded_areas_on_PD_hist(ax, pd_locations, bins, shaded_at, shade_color='r', shade_alpha=.2):
	counts, _, _ = ax.hist(pd_locations, bins=bins, color='0')
	ax.set_xlabel("Location (m)")
	ax.set_ylabel("Number of PDs")
    
	ax.fill_between(bins[:-1], 0, np.max(counts), where=shaded_at, color=shade_color, alpha=shade_alpha)


def inside_a_cluster(location, clusters, bin_size=4.0):
	bin_index = int(location/bin_size)
	return any(a <= bin_index < b for a,b in clusters)

def show_plot(locations, times, charges, bins, clusters, circ):
	plt.rcParams['figure.figsize'] = [8, 4]
	inside_a_cluster_v = np.vectorize(lambda location: inside_a_cluster(location, clusters))
	fig, ax = plt.subplots(2,1,figsize=(8,8))
	ax1,ax2 = ax
	draw_shaded_areas_on_PD_hist(ax1, locations, bins, inside_a_cluster_v(bins[:-1]), 'r', .3)
	ax1.set_title("The {0} found clusters".format(len(clusters)))
	ax2.scatter(x=locations, y=times, s=charges/1e4, c='.0')
	ax2.set_xlabel("Location")
	ax2.set_ylabel("Date")
	ax2.set_title("Circuit {0}".format(circ))
	ax2.fill_between(bins[:-1], np.min(times), np.max(times), where=inside_a_cluster_v(bins[:-1]), color='r', alpha=.1)
	fig.subplots_adjust(hspace=.4)
	fig.show()
