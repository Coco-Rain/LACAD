import cadquery as cq

result = (
cq.Workplane("XY")
.text("Hello", fontsize=5, distance=1)
)
cq.exporters.export(result, 'GT.stl')