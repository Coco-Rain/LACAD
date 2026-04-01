import cadquery as cq

result = (
cq.Workplane("XY")
.ellipse(6, 4)
.extrude(2)
)
cq.exporters.export(result, 'GT.stl')