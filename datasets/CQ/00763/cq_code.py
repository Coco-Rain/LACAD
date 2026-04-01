import cadquery as cq

result = (
cq.Workplane("XY")
.box(30, 25, 8)
.faces(">Z")
.workplane()
.ellipse(10, 5)
.extrude(6)
.faces(">Z")
.workplane()
.hole(4)
)
cq.exporters.export(result, 'GT.stl')