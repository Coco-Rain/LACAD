import cadquery as cq

result = (
cq.Workplane("XY")
.rect(10, 10)
.tag("outer")
.rect(5, 5)
.tag("inner")
.wires(tag="inner")
.extrude(3)
)
cq.exporters.export(result, 'GT.stl')