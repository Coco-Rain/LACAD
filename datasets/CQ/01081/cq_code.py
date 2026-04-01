import cadquery as cq

result = (
cq.Workplane("XY")
.moveTo(0, 0)
.hLineTo(5)
.vLineTo(4, True)
.moveTo(2, 0)
.vLineTo(3)
.hLineTo(4)
.vLineTo(0)
.close()
.extrude(1.5)
)
cq.exporters.export(result, 'GT.stl')