import cadquery as cq

result = (
cq.Workplane("XY", (0, 0, 13))
.moveTo(-60., 0)
.line(0, 3)
.line(25, 0)
.line(10, 7)
.line(50, 0)
.line(10, -7)
.line(25, 0)
.line(0, -3)
.close()
.revolve(axisEnd=(1, 0))
)
cq.exporters.export(result, 'GT.stl')