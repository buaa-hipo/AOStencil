def stencil_3d7pt_star(lattice:[128,128,130,'double']):
  for z in range(1,127):
    for y in range(1,127):
      for x in range(1,129):
        lattice[t+1,z,y,x]=0.1*lattice[t,z+1,y,x]+0.2*lattice[t,z-1,y,x]+0.3*lattice[t,z,y-1,x]+0.4*lattice[t,z,y+1,x]+0.5*lattice[t,z,y,x-1]+0.6*lattice[t,z,y,x+1]+0.7*lattice[t,z,y,x]
