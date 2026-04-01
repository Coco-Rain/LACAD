import cadquery as cq

result = (
cq.Workplane("XY")
.sketch()
.rect(20, 10)
.finalize()
.extrude(5)
)
cq.exporters.export(result, 'GT.stl')