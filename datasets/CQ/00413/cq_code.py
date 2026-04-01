import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 30, 5)
.faces(">Z").workplane()
.box(15, 15, 5)
.faces(">Z").workplane()
.box(7, 7, 5)
.end(2)
)
cq.exporters.export(result, 'GT.stl')