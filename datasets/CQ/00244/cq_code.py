import cadquery as cq

result = (
cq.Workplane("XY")
.box(40, 40, 5)
.faces(">Z")
.workplane()
.rarray(10, 10, 4, 4)
.rect(3, 3)
.extrude(2)
)
cq.exporters.export(result, 'GT.stl')