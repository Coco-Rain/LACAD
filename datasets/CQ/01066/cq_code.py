import cadquery as cq

path = (
cq.Workplane("XZ")
.spline([(0,0), (2,3), (5,4)])
.wire(forConstruction=True)
)
result = (
cq.Workplane("XY")
.circle(1.5)
.sweep(path)
)
cq.exporters.export(result, 'GT.stl')