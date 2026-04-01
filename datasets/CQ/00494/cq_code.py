import cadquery as cq

result = (
cq.Workplane("XY")
.sphere(5)
.transformed(offset=(0, 0, 10))
.cylinder(3, 10)
.solids()
)
cq.exporters.export(result, 'GT.stl')