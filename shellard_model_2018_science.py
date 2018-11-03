####
# Python script to simulate collective cell chemotaxis due to inhibited front contractility of a supracellular
# actomyosin ring present in the peripheral cells of a cell cluster. This script is part of a study by 
# Adam Shellard, Andras Szabo, Xavier Trepat, and Roberto Mayor, published in Science, 2018. If you are 
# using this code, please cite the original publication:
# 
# Adam Shellard, András Szabó, Xavier Trepat, Roberto Mayor. Supracellular contraction at the rear of neural crest cell groups drives collective chemotaxis. Science 362(6412): 339-343, 2018. DOI: 10.1126/science.aau3301 
#
# This script is written by Andras Szabo (andras.szabo.81@gmail.com). 
# 
# Main dependencies used for running the script:
# python 2.7.9
# scipy 1.1.0  
# numpy 1.15.3  
# matplotlib 2.2.3      
# 
####


import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from collections import defaultdict

####--------------------------------------------------------------------------------------####
# Parameters of the model

# --------------------------------------------------- # Geometry of the simulation
global_initial_areas	= np.array([ [130,130], [270,270] ])	# initiate cells in these ranges: from (x,y) -> to (x,y)
global_Lmax		= np.array([400, 600])			# size of area from [0,0] to this position
global_N		= 60					# number of cells

# --------------------------------------------------- # General simulation parameters
global_Tmax		= 10000					# maximum time of simulations
global_Tout		= 20					# time-frames for output
global_dt		= 0.1					# time-step for Euler iterations
global_full_contraction = False					# set isotropic contraction, otherwise back half contracts
global_intercalate_until= global_Tmax				# allow neighbour exchange until this time
####--------------------------------------------------------------------------------------####

# --------------------------------------------------- # Cell size and other lengths, radii
global_ranges 		= [6., 9., 9.2, 11.]			# repulsion core, shortest adhesion distance, 
								# maximum adhesion force distance, furthest 
								# reach of the cell

# --------------------------------------------------- # Other cellular properties
global_speed		= 2					# target speed of cells
global_velDampening	= 0.1					# velocity dampening, or friction of movement (less than 1)
global_contraction_time	= 100					# minimum duration of contractions
global_relaxation_time	= 100					# minimum time between contractions

# --------------------------------------------------- #	Interaction strengths (negative = attraction)
global_I_rnd 		= 5					# random movement strength of cell
global_I_spp		= 5					# self-propusion strength of cell
global_I_rep 		= 10					# core cell repulsion strength
global_I_contact	= -0.5					# cell-cell adhesion strength
global_I_contraction	= -5					# actomyosin contraction strength


class Cell(object):
	type = 0	# cell type is 0 for peripheral cells and 1 for internal cells
	pos = np.array((2))
	newpos = np.array((2))
	vel = np.array((2))
	newvel = np.array((2))
	acc = np.array((2))
	
	pol = np.asarray([0.0, 0.0])
	pol_age = 0

	speed = 0
	velDampening = 0
	
	steric_radius = 0
	neutral_radius = 0
	adhesion_radius = 0
	reach_radius = 0
	
	steric_coeff = 0
	adhesion_coeff = 0
	speed_coeff = 0
	rnd_coeff = 0
	
	steric_gradient = 0
	adhesion_gradientA = 0
	adhesion_gradientB = 0

	actomyosin_contraction = False
	actomyosin_off_time = 0
	actomyosin_on_time = 0

	neighbor = []
	
	
	def __init__(self, pos, type, I, R, props, N):
		self.pos = np.asarray(pos)
		self.newpos = np.asarray(pos)
		vel = np.array([0,0])
		self.vel = np.asarray(vel)
		self.newvel = np.asarray(vel)
		self.type = type
		
		self.speed = props[0]
		self.velDampening = props[1]

		self.steric_coeff = I[0]
		self.adhesion_coeff = I[1]
		self.speed_coeff = I[2]
		self.rnd_coeff = I[3]
		
		self.reset_ranges(R)

	def reset_ranges(self, new_ranges):
		# Re-adjust the ranges of adhesion and repulsion interactions

		self.steric_radius, self.neutral_radius, self.adhesion_radius, self.reach_radius = \
			new_ranges[0], new_ranges[1], new_ranges[2], new_ranges[3]

		self.steric_gradient = self.steric_coeff / self.steric_radius
		self.adhesion_gradientA = self.adhesion_coeff / (self.adhesion_radius - self.neutral_radius)
		self.adhesion_gradientB = self.adhesion_coeff / (self.reach_radius - self.adhesion_radius)

			
	def actomyosin_contraction_on(self):
		ret = False

		if self.actomyosin_off_time >= global_relaxation_time:
			self.actomyosin_contraction = True
			ret = True
		return ret

	def actomyosin_contraction_off(self):
		ret = False

		if self.actomyosin_on_time >= global_contraction_time:
			self.actomyosin_contraction = False
			ret = True
		return ret


	def plot_colour(self):
		# return the colour to be used for plotting this cell
		
		if self.type == 0:
			col = [0.5, 0.5, 0.6]		# Default cell colour
			if self.actomyosin_contraction:
				col = [0.8, 0.75, 0.1]	# Contracting cell colour
		if self.type == 1:
			col = [0.2, 0.61, 0.35]
		return col
	

def unit_vector(v):
	# Return the unit vector of the 2D vector v
	length = np.linalg.norm(v)
	if length > 0:
		return (v / length)
	else:
		return 0 * v


def initiateCells(N, areas, I, R, props):
	
	# Initiate cells in a pre-set geometry

	cell = []
	type = 1
	Ntotal = np.sum(N)
	id = 0
	
	centre = np.asarray([(areas[1][0] + areas[0][0]) * 0.5, (areas[1][1] + areas[0][1]) * 0.5])
	radius = (areas[1][0] - areas[0][0]) * 0.5
	dx = areas[1][0] - areas[0][0]
	dy = areas[1][1] - areas[0][1]
	xmin = areas[0][0]
	ymin = areas[0][1]
	for counter in xrange(N):
		pos = np.asarray([0.0,0.0])
		too_close = True
		tries = 0 
		while (too_close):
			pos = np.asarray([random.random() *dx + xmin, random.random() *dy + ymin])
			if (np.linalg.norm(pos-centre) < radius) : 
				mindist=-1
				for i in xrange(id):
					dist = np.linalg.norm( pos - cell[i].pos )
					if mindist < 0 or mindist > dist:
						mindist = dist
				if mindist <0 or mindist > 2.0 * cell[i].steric_radius:
					too_close=False
				else:
					if tries > 500:
						print "Too high density, failed to initialize"
						exit()
			tries += 1
			if tries > 500:
				print "Too high density, failed to initialize"
				exit()
		c = Cell(pos, type, I, R, props, Ntotal)
		cell.append(c)
		id += 1

	return cell

			
def iterate(cell, dt, time):
	# Iterate the position of all cells 

	for id in xrange(len(cell)):

		#--- Components controlling cell displacement / movement ---#
		# ++ Interaction with neighbour cells (neighbours are determined through the update_cells function)
		f_interaction = np.asarray([0.0,0.0])
		for nid in cell[id].neighbor:
			if (id != nid):
				f, dist = interact(cell, id, nid)
				f_interaction += f

		# ++ Self-propulsion
		f_spp = cell[id].speed_coeff * (cell[id].speed - np.linalg.norm(cell[id].vel)) * unit_vector(cell[id].vel)
		
		# ++ Noise
		f_noise = cell[id].rnd_coeff * (np.asarray([random.random(), random.random()]) - 0.5)
		
		
		
		# Cell movement dynamics: 'friction' + interaction + spp + noise 
		cell[id].newvel = dt * ((1.0 - cell[id].velDampening) * cell[id].vel + f_interaction + f_spp + f_noise)
		cell[id].newpos = cell[id].pos + dt * cell[id].vel

		# Closed boundary conditions: cannot go into walls
		if cell[id].newpos[0] < 0:		# left 
			if cell[id].newvel[0] < 0:
				cell[id].newvel[0] = 0
		if cell[id].newpos[0] > global_Lmax[0]:	# right
			if cell[id].newvel[0] > 0:
				cell[id].newvel[0] = 0
		if cell[id].newpos[1] < 0: 		# top
			if cell[id].newvel[1] < 0:
				cell[id].newvel[1] = 0
		if cell[id].newpos[1] > global_Lmax[1]:	# bottom
			if cell[id].newvel[1] > 0:
				cell[id].newvel[1] = 0
		#----

		
	# Update cell positions
	for cellid in xrange(len(cell)):
		cell[cellid].pos = cell[cellid].newpos
		cell[cellid].vel = cell[cellid].newvel

	return cell


def interact(cell, cellid, ncellid):
	# Interaction of cell 'cellid' and cell 'ncellid'. 
	# Cell information is stored in the cell object
	# Return the effect (or "force") on cell a. 
	
	f = np.asarray([0.0,0.0])
	m = 0 
	dist = np.linalg.norm(cell[cellid].pos - cell[ncellid].pos)
	dir = unit_vector(cell[cellid].pos - cell[ncellid].pos)
	
	### The interaction "potential" ###

	if dist < (cell[cellid].reach_radius + cell[ncellid].reach_radius):
		# Only interact if within range

		if dist < (cell[cellid].steric_radius + cell[ncellid].steric_radius):
			# repell:
			delta = cell[cellid].steric_radius + cell[ncellid].steric_radius - dist 
			m += cell[cellid].steric_gradient * delta

		elif dist < (cell[cellid].neutral_radius + cell[ncellid].neutral_radius):
			# cells are in the neutral region, no effect on movement
			m += 0

		elif dist < (cell[cellid].adhesion_radius + cell[ncellid].adhesion_radius):
			# attract
			delta = dist - (cell[cellid].neutral_radius + cell[ncellid].neutral_radius)
			m += cell[cellid].adhesion_gradientA * delta 

		elif dist < (cell[cellid].reach_radius + cell[ncellid].reach_radius):
			# attract
			delta = dist - (cell[cellid].adhesion_radius + cell[ncellid].adhesion_radius)
			m += cell[cellid].adhesion_gradientB * delta 

		# Special interaction between peripheral cells
		if cell[cellid].type == 0 and cell[ncellid].type == 0:

			# If both cells contract, add myosin force:
			if cell[cellid].actomyosin_contraction and cell[ncellid].actomyosin_contraction: 
				m += global_I_contraction	# Use a constant attraction force

	f = m * dir
	return f, dist


def set_peripheral_or_central_cell_type(cell):
	# Decide for each cell whether it is at the periphery (type=0) or internal to a cluster (type=1)

	# The angle between neighbouring connections above which the cell is considered to be at the periphery:
	threshold = np.pi * 0.75
	
	# 1. Determine neighbours
	if len(cell) > 2:
		# Use Voronoi / Delaunay tessellation to infer neighbours:
		pos = np.zeros((len(cell), 2))
		for id in xrange(len(cell)):		# Collect the positions of the cells in an array
			pos[id][0] = cell[id].pos[0]
			pos[id][1] = cell[id].pos[1]
		tessellation = Delaunay(pos)

		# Determine neighbours of all cells:
		neighbors = defaultdict(set)
		for simplex in tessellation.vertices:
			for idx in simplex:
				other = set(simplex)
				other.remove(idx)
				neighbors[idx] = neighbors[idx].union(other)

				# Only those cells count as neighbours, which are within reach:
				cell[idx].neighbor = []
				for nid in neighbors[idx]:
					if (np.linalg.norm( cell[idx].pos - cell[nid].pos) < \
					   (cell[idx].reach_radius + cell[nid].reach_radius)):
						# cells are within reach of each other
						cell[idx].neighbor.append(nid)
	else:
		# For two cells only, neighbours can be determined simply based on distance
		if len(cell) == 2:
			if np.linalg.norm(cell[0].pos - cell[1].pos) < cell[0].reach_radius + cell[1].reach_radius:
				cell[0].neighbor = [1]
				cell[1].neighbor = [0]
			else:
				cell[0].neighbor = []
				cell[1].neighbor = []
		else:
			cell[0].neighbor = []
			

	# 2. Based on the neighbours, determine if each cell is at the periphery (0) or centre (1):
	for id in xrange(len(cell)):
		nxs = []
		nys = []
		nids = []

		for nid in cell[id].neighbor:
			nxs.append(cell[nid].pos[0] - cell[id].pos[0])
			nys.append(cell[nid].pos[1] - cell[id].pos[1])
			nids.append(nid)
		
		nangles = np.sort(np.arctan2(nys, nxs))
		diffangles = []

		# detect if there is a gap between two consecutive angles that are
		# more than 'threshold' apart --> this will be considered as a free edge (type = 0)
		if len(nangles) > 3:
			type = 1
			for a in xrange(len(nangles)-1):
				diffangles.append(nangles[a+1] - nangles[a])
			diffangles.append(nangles[0] + 2.0*np.pi - nangles[len(nangles)-1])
			
			if np.max(diffangles) > threshold:
				type = 0
		else:
			type = 0
		cell[id].type = type

def update_cell_dynamics (cell, dt):
	# Advance any dynamics in the cells

	for id in xrange(len(cell)):
		if cell[id].actomyosin_contraction:
			cell[id].actomyosin_on_time += dt
			cell[id].actomyosin_off_time = 0
		else:
			cell[id].actomyosin_on_time   = 0
			cell[id].actomyosin_off_time += dt


def update_cells (cell, iteration, time, dt, counter):
	# Apply any manipluation to the cells here if needed
	# time is the number of iteration step rather than 'real time'
	
	
	# Decide the type of the cell: peripherial (0) or internal (1)
	if time <= global_intercalate_until :
		set_peripheral_or_central_cell_type(cell)
	update_cell_dynamics(cell, dt)
	counter += dt
	if counter > global_contraction_time:
		counter = -global_relaxation_time
	

	if (iteration % 10 and time > 100):
		pxs, pys = [], []
		for id in xrange(len(cell)):
			pxs.append(cell[id].pos[0])
			pys.append(cell[id].pos[1])
		cmx, cmy = np.mean(pxs), np.mean(pys)
		sizex = np.max(pxs) - np.min(pxs)
		sizey = np.max(pys) - np.min(pys)

		# Update actomyosin contaction properties:
		for id in xrange(len(cell)):
			if cell[id].type == 0:
				# only perturb the properties of the cells at the periphery
				
				# determine positional direction of the cell with respect to the centre of mass
				if global_full_contraction or time < 200:
					# uniform contractions before the initial period and in isotropic contraction 
					# simulations; set direction to back (along positive y-axis)
					direction = np.arctan2(1,0)
				else: 
					# switch on and off back cells together periodically:
					direction = np.arctan2(cmy - cell[id].pos[1], cmx - cell[id].pos[0])
				
				# contract at the back but not the front:
				if direction > np.pi * (-8.0/8.0) and direction < np.pi * (-0.0/8.0):
					# At the front, don't turn on
					cell[id].actomyosin_contraction = False
				else:
					# At the back, consider contracting
					if counter > 0:
						cell[id].actomyosin_contraction = True
					else:
						cell[id].actomyosin_contraction = False
	return counter
					

def output_state(cell, t, Lmax):
	# Output an image frame and print the coordinates to stdout

	# Plot:
	fig = plt.gcf()
	ax = plt.gca()
	ax.cla()
	# Uncomment to draw grid:
	#ax.grid(b=True, which='major')
	# Draw boundaries
	fig.gca().add_artist(plt.Rectangle((0,0),Lmax[0],Lmax[1],fill=False, color='0.75'))

	# Draw cells
	for i in xrange(len(cell)):
		col = cell[i].plot_colour()
		s = cell[i].steric_radius
		if s == 0:
			s=2
		cx=cell[i].pos[0]
		cy=cell[i].pos[1]
		fig.gca().add_artist(plt.Circle((cx,cy),s,color=col,fill=True))

	bufferx, buffery = 0.25*Lmax[0], 0.25*Lmax[1]
	if bufferx < 10:
		bufferx = 10
	if buffery < 10:
		buffery = 10
	if bufferx > 100:
		bufferx = 100
	if buffery > 100:
		buffery = 100
	# Add label
	fig.gca().add_artist(plt.Text(x=(Lmax[0]*0.75),y=(Lmax[1]+buffery*0.5), text=str(t), \
		horizontalalignment='right', size='16'))
	ax.set_xlim(-bufferx, Lmax[0]+bufferx)
	ax.set_ylim(-buffery, Lmax[1]+buffery)
	ax.set_aspect(1)
	ax.invert_yaxis()
	fig.savefig('config-'+"{0:05d}".format(int(t))+'.png', dpi=150, bbox_inches='tight')

	# Print coordinates
	for id in xrange(len(cell)):
		contraction = 0
		if cell[id].actomyosin_contraction:
			contraction = 1
		print t, cell[id].type, id, cell[id].pos[0], cell[id].pos[1], contraction
	print "\n"


def main():
	cellsize = global_ranges[0]
	N = global_N
	Lmax = global_Lmax
	Tmax = global_Tmax
	t_out = global_Tout
	dt = global_dt
	counter = -1.0 * global_relaxation_time	
	
	# Read coefficients:
	interaction_coeffs = np.transpose(np.array([global_I_rep, global_I_contact, global_I_spp, \
		global_I_rnd]))
	spatial_ranges = np.transpose(np.array(global_ranges))
	props = np.transpose(np.array([global_speed, global_velDampening]))


	# Initiate cells:
	cell = initiateCells(N, global_initial_areas, interaction_coeffs, \
		spatial_ranges, props)

	# Save initial configuration and start simulation
	output_state (cell, 0.0, Lmax)
	for it in xrange(1,int(Tmax / dt)+1):
		t = it * dt		
		counter = update_cells(cell, it, t, dt, counter)
		iterate(cell, dt, t) 
		if (t % t_out) == 0 :
			output_state(cell, t, Lmax)
	
	

if __name__ == "__main__":
	main()
