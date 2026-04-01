import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(0, 10)
.polarLine(10, 45)
.polarLine(8, -30)
.close()
.extrude(4)
)
cq.exporters.export(result, 'GT.stl')