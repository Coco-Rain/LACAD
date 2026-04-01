import cadquery as cq

result = (
cq.Workplane("XY")
.polygon(6, 20)
.extrude(10)
.faces(">Z")
.vertices()
.tag("top_vertices")
.workplaneFromTagged("top_vertices")
.circle(1.5)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')