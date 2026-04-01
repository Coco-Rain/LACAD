import cadquery as cq

result = (
cq.Workplane()
.rect(5, 10)
.extrude(4)
.faces(">Z")
.workplane()
.center(5, 0)
.rect(5, 10)
.extrude(4)
)
cq.exporters.export(result, 'GT.stl')