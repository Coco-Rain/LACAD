import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 20, 5)
.faces(">Z").workplane()
.box(10, 10, 5)
.end()
.transformed(offset=(0, 0, 10)).box(15, 15, 5)
)
cq.exporters.export(result, 'GT.stl')