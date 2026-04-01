import cadquery as cq

result = (
cq.Workplane("XY")
.rect(10, 10)
.extrude(1)
)
svg_output = result.toSvg()
print(svg_output)
cq.exporters.export(result, 'GT.stl')