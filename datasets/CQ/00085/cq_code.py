import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(0, 0)
.lineTo(8, 0)
.threePointArc((8, 4), (4, 8))
.lineTo(0, 4)
.threePointArc((4, 0), (0, 0))
.close()
.extrude(6)
)
cq.exporters.export(result, 'GT.stl')