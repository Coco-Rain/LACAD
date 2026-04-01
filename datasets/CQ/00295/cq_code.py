import cadquery as cq

result = (
cq.Workplane("XY")
.box(15, 15, 5)
.faces(">Z")
.workplane()
.center(3, 3)
.hole(4)
)
cq.exporters.export(result, 'GT.stl')