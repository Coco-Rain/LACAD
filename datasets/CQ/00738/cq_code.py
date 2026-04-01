import cadquery as cq

result = (
cq.Workplane("XZ")
.moveTo(0, 5)
.lineTo(3, 5)
.lineTo(5, 0)
.close()
.revolve(angleDegrees=270,
axisStart=(0, 0, 0),
axisEnd=(0, 0, 1))
)
cq.exporters.export(result, 'GT.stl')