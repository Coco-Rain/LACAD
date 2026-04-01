import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(2, 1)
.hLineTo(8)
.vLineTo(5)
.hLineTo(2)
.close()
.extrude(3)
)
cq.exporters.export(result, 'GT.stl')