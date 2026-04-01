import cadquery as cq

result = (
cq.Workplane("XY")
.polygon(nSides=6, diameter=20)
.extrude(5)
.faces(">Z")
.workplane()
.moveTo(8, 0)
.radiusArc((0, 8), 10)
.lineTo(0, 0)
.close()
.cutBlind(-3)
)
cq.exporters.export(result, 'GT.stl')