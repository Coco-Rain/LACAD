import cadquery as cq

def hyperbolic_paraboloid(u, v):
x = 8 * (u - 0.5)
y = 8 * (v - 0.5)
z = 0.5 * (x**2 - y**2)
return (x, y, z)
result = (
cq.Workplane("XY")
.parametricSurface(hyperbolic_paraboloid, N=25)
)
cq.exporters.export(result, 'GT.stl')