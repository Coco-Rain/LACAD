import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(1, 1)
.hLineTo(4)
.vLineTo(3)
.hLineTo(1)
.vLineTo(1)
.close()
.extrude(2)
)
cq.exporters.export(result, 'GT.stl')