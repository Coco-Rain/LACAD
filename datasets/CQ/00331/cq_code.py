import cadquery as cq

result = (
cq.Workplane("XY")
.rect(20, 10)
.extrude(5)
.faces(">Z")
.workplane()
.center(-3, -3)
.hole(2)
)
cq.exporters.export(result, 'GT.stl')