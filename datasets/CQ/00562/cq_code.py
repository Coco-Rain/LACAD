import cadquery as cq

result = (
cq.Workplane("XY")
.sketch()
.slot(10, 2, angle=0)
.finalize()
.extrude(4)
)
cq.exporters.export(result, 'GT.stl')