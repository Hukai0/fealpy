
import fealpy.cgraph as cgraph

WORLD_GRAPH = cgraph.WORLD_GRAPH
mesher = cgraph.create("InpMeshReader") 
matrixer = cgraph.create("MatMatrixReader")              
spacer = cgraph.create("TensorFunctionSpace")        
eig_eq = cgraph.create("GearBox")
eigensolver = cgraph.create("SLEPc")
mesher(input_inp_file ='/home/cbtxs/Downloads/box_case3.inp') 
matrixer(input_mat_file ='/home/cbtxs/Downloads/shaft_case3.mat') 
spacer(mesh = mesher(), gd = 3)
eig_eq(mesh=mesher(),shaftmatrix = matrixer(), space=spacer(),q = 3)


eigensolver(
    S=eig_eq().stiffness,
    M=eig_eq().mass,
    neigen=6,
)

WORLD_GRAPH.output(eig_eq=eigensolver().val, uh=eigensolver().vec)
WORLD_GRAPH.error_listeners.append(print)
WORLD_GRAPH.execute()
print(WORLD_GRAPH.get())