import cadquery as cq

result = (
cq.Workplane("XY")
.box(25, 18, 12)
.union(
cq.Workplane("XY")
.transformed(offset=(0, 0, 6))
.cylinder(9, 5)
)
.faces(">Z")
.workplane()
.hole(6, 8)
)
cq.exporters.export(result, 'GT.stl')