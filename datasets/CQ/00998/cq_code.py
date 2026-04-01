import cadquery as cq

result = (
cq.Workplane("YZ")
.circle(5)
.extrude(10)
)
svg_output = result.toSvg()
print(svg_output)
cq.exporters.export(result, 'GT.stl')