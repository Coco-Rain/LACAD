import cadquery as cq

result = (
cq.Workplane("XY")
.box(15, 15, 3)
.faces(">Z")
.workplane()
.rect(10, 5)
.extrude(2)
)
cq.exporters.export(result, 'GT.stl')