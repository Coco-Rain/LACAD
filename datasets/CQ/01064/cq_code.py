import cadquery as cq

result = (
cq.Workplane("XY")
.sketch()
.rect(8, 6)
.finalize()
.extrude(3)
)
cq.exporters.export(result, 'GT.stl')