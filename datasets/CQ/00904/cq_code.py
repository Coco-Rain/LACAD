import cadquery as cq

result = (
cq.Workplane("XY")
.cylinder(10, 5)
.transformed(offset=(0, 0, 10))
.box(6, 6, 2)
.solids()
)
cq.exporters.export(result, 'GT.stl')