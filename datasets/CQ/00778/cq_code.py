import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(5, 0)
.lineTo(5, 5)
.lineTo(0, 5)
.lineTo(-5, 0)
.lineTo(0, -5)
.close()
.extrude(2)
)
cq.exporters.export(result, 'GT.stl')