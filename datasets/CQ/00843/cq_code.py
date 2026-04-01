import cadquery as cq

result = (
cq.Workplane("XY")
.sketch()
.rect(30, 15, angle=45)
.finalize()
.extrude(10)
)
cq.exporters.export(result, 'GT.stl')