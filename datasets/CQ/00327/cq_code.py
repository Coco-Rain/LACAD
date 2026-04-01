import cadquery as cq

result = (
cq.Workplane("YZ")
.text("CadQuery", fontsize=10, distance=2)
)
cq.exporters.export(result, 'GT.stl')