import cadquery as cq

result = (
cq.Workplane("XY")
.sketch()
.rect(10, 10)
.rect(5, 5)
.clean()
.finalize()
.extrude(1)
)
cq.exporters.export(result, 'GT.stl')