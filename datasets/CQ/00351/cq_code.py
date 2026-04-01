import cadquery as cq

result = (
cq.Workplane("XY")
.sketch()
.rect(8, 6)
.vertices()
.chamfer(1.5)
.finalize()
.extrude(3)
)
cq.exporters.export(result, 'GT.stl')