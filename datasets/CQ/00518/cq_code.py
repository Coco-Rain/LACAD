import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(0, 0)
.lineTo(20, 0)
.threePointArc((25, 5), (20, 10))
.lineTo(0, 10)
.close()
.extrude(5)
.faces(">Z")
.workplane()
.rect(8, 6)
.cutBlind(-3)
)
cq.exporters.export(result, 'GT.stl')