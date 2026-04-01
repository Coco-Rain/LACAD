import cadquery as cq

result = (
cq.Workplane("XY")
.sketch()
.ellipse(5, 3)
.finalize()
.extrude(2)
)
cq.exporters.export(result, 'GT.stl')