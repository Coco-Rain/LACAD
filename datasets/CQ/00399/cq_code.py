import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(0, 0)
.lineTo(5, 0)
.tangentArcPoint((5, 5))
.lineTo(0, 5)
.close()
.extrude(3)
)
cq.exporters.export(result, 'GT.stl')