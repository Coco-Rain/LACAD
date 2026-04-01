import cadquery as cq

result = (
cq.Workplane("XY")
.sketch()
.rect(12, 8)
.reset()
.edges()
.fillet(1.5)
.finalize()
.extrude(6)
)
cq.exporters.export(result, 'GT.stl')