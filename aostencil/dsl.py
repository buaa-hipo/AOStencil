import ast
from .stencil import Stencil2dIR,Stencil3dIR

try:
    import numpy as np
except ImportError:
    from . import my_array as np

class StencilVisitor(ast.NodeVisitor):
    def __init__(self):
        self.name = None
        self.lattice_shape = None
        self.x_range = None
        self.y_range = None
        self.z_range = None
        self.offset2stencil = None
        self.datatype = None

    def convert_stencil_to_map(self,assign_node):
        stencil_map = {}
        
        if not isinstance(assign_node, ast.Assign):
            raise ValueError("The provided node is not an Assign node.")
        
        if len(assign_node.targets) != 1 or not isinstance(assign_node.targets[0], ast.Subscript):
            raise ValueError("The target of the assignment is not a valid Subscript.")
        
        target_subscript = assign_node.targets[0]
        
        if not isinstance(target_subscript.slice.value, ast.Tuple):
            raise ValueError("The slice of the subscript is not a Tuple.")
        
        def extract_offset(element):
            if isinstance(element, ast.BinOp):
                if isinstance(element.op, ast.Sub):
                    return -element.right.n
                elif isinstance(element.op, ast.Add):
                    return element.right.n
            elif isinstance(element, ast.Name):
                return 0
            elif isinstance(element, ast.Num):
                return element.n
            raise ValueError("Unexpected element in offset extraction.")
        
        def parse_binop(binop):
            coefficient = binop.left.n
            subscript = binop.right
            index = subscript.slice.value.elts[1:]  # Skip the time index
            relative_index = tuple(extract_offset(idx) for idx in index)[::-1]
            return relative_index, coefficient
        
        def extract_coefficients(binop):
            if isinstance(binop, ast.BinOp) and isinstance(binop.op, ast.Add):
                left_offsets, left_coefficients = extract_coefficients(binop.left)
                right_offsets, right_coefficients = extract_coefficients(binop.right)
                return left_offsets + right_offsets, left_coefficients + right_coefficients
            elif isinstance(binop, ast.BinOp) and isinstance(binop.op, ast.Mult):
                offset, coefficient = parse_binop(binop)
                return [offset], [coefficient]
            else:
                raise ValueError("Unexpected element in coefficient extraction.")
        
        offsets, coefficients = extract_coefficients(assign_node.value)
        stencil_map.update(dict(zip(offsets, coefficients)))
        
        return stencil_map

    def visit_FunctionDef(self, node):
        self.name = node.name
        
        # Assuming the lattice argument is the first argument
        if node.args.args:
            lattice_arg = node.args.args[0]
            
            if isinstance(lattice_arg.annotation, ast.List):
                self.lattice_shape = [elt.n for elt in lattice_arg.annotation.elts[:-1]]
                self.datatype = lattice_arg.annotation.elts[-1].s if isinstance(lattice_arg.annotation.elts[-1], ast.Str) else lattice_arg.annotation.elts[-1].id
            elif isinstance(lattice_arg.annotation, ast.Subscript) and isinstance(lattice_arg.annotation.value, ast.Name):
                if lattice_arg.annotation.value.id == 'List':
                    self.lattice_shape = [elt.n for elt in lattice_arg.annotation.slice.value.elts[:-1]]
                    self.datatype = lattice_arg.annotation.slice.value.elts[-1].s if isinstance(lattice_arg.annotation.slice.value.elts[-1], ast.Str) else lattice_arg.annotation.slice.value.elts[-1].id
        
        # Traverse the body of the function to find loops and assignments
        self.find_loops_and_assignments(node.body)
        self.generic_visit(node)


    def find_loops_and_assignments(self, body):
        for stmt in body:
            if isinstance(stmt, ast.For):
                loop_var = stmt.target.id
                loop_range = self.get_range(stmt.iter)
                
                if loop_var == 't':
                    continue
                elif loop_var == 'x':
                    self.x_range = loop_range
                elif loop_var == 'y':
                    self.y_range = loop_range
                elif loop_var == 'z':
                    self.z_range = loop_range
                
                # Continue to find nested loops and assignments
                self.find_loops_and_assignments(stmt.body)
            elif isinstance(stmt, ast.Assign):
                # Update the stencil with the latest assignment to lattice
                if isinstance(stmt.targets[0], ast.Subscript):
                    subscript = stmt.targets[0]
                    if isinstance(subscript.value, ast.Name) and subscript.value.id == 'lattice':
                        self.offset2stencil = self.convert_stencil_to_map(stmt)

    def get_range(self, iter_node):
        if isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name) and iter_node.func.id == 'range':
            args = iter_node.args
            if len(args) == 1:
                return (0, args[0].n)  # range(n) -> (0, n)
            elif len(args) == 2:
                return (args[0].n, args[1].n)  # range(start, stop) -> (start, stop)
            elif len(args) == 3:
                return (args[0].n, args[1].n, args[2].n)  # range(start, stop, step) -> (start, stop, step)
        return None

    def to_stencil(self):
        assert(self.name!=None)
        s=None
        if self.z_range==None:
            assert(self.x_range[0]+self.x_range[-1]==self.lattice_shape[1])
            assert(self.y_range[0]+self.y_range[-1]==self.lattice_shape[0])

            stencil_coefficient=np.zeros((self.y_range[0]*2+1,self.x_range[0]*2+1))

            for offset,val in self.offset2stencil.items():
                try:
                    stencil_coefficient[offset[1]+self.y_range[0],offset[0]+self.x_range[0]]=val
                except IndexError:
                    raise IndexError("stencil kernel is not match to lattice edge")

            s=Stencil2dIR(self.lattice_shape[0],self.lattice_shape[1],stencil_coefficient,0,self.datatype)

        else:
            assert(self.x_range[0]+self.x_range[-1]==self.lattice_shape[2])
            assert(self.y_range[0]+self.y_range[-1]==self.lattice_shape[1])
            assert(self.z_range[0]+self.z_range[-1]==self.lattice_shape[0])

            stencil_coefficient=np.zeros((self.z_range[0]*2+1,self.y_range[0]*2+1,self.x_range[0]*2+1))

            for offset,val in self.offset2stencil.items():
                try:
                    stencil_coefficient[offset[2]+self.z_range[0],offset[1]+self.y_range[0],offset[0]+self.x_range[0]]=val
                except IndexError:
                    raise IndexError("stencil kernel is not match to lattice edge")

            s=Stencil3dIR(self.lattice_shape[0],self.lattice_shape[1],self.lattice_shape[2],stencil_coefficient,0,self.datatype)
        
        s.set_name(self.name)
        return s


def from_dsl_load_stencil(source_code):
    parsed_ast = ast.parse(source_code)
    visitor = StencilVisitor()
    visitor.visit(parsed_ast)
    return visitor.to_stencil()



