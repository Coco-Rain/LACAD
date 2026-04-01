import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(5, 0)
.polarLineTo(10, 60)
.polarLineTo(10, -60)
.close()
.extrude(4)
)
cq.exporters.export(result, 'GT.stl')